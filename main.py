import json
import logging, traceback
import asyncio
from typing import Annotated, Literal
from fastapi import FastAPI, Depends, Query, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import asc, desc
from datetime import datetime
from utils import validate_date_format, validate_date_range, calculate_months_between
from openai import OpenAI
client = OpenAI()

# 1. Define a list of callable tools for the model
tools = [
    {
        "type": "function",
        "name": "get_balance",
        "description": "Fetch balance for a specific month. If the user asked a relative period (e.g., 'two months ago', 'last month'), convert that to a concrete YYYY-MM first and pass it here.",
        "strict": "true",
        "parameters": {
            "type": "object",
            "properties": {
                "month": {
                    "type": "string",
                    "description": "Target month in YYYY-MM (e.g., '2025-07'). If omitted, backend uses current month."
                }
            }
        }
    }
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # This ensures logs are printed to console
    ]
)
logger = logging.getLogger('app')

app = FastAPI()

# Global exception handler for 500 errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler that catches all unhandled exceptions,
    logs them with stack trace, and returns a 500 error response.
    """
    # Log the exception with full stack trace
    logger.error(
        f"Unhandled exception occurred: {type(exc).__name__}: {str(exc)}\n"
        f"Request: {request.method} {request.url}\n"
        f"Stack trace:\n{traceback.format_exc()}"
    )

    # Return a generic 500 error response
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_type": type(exc).__name__
        }
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://finance-app-three-rosy.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class Balance(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    date: str = Field(index=True)
    balance: float
    

# I/O models
class BalanceBase(SQLModel):
    date: str
    balance: float


class BalanceCreate(BalanceBase):
    pass


class BalanceRead(BalanceBase):
    id: int

class BalanceWithDelta(BalanceRead):
    delta: float | None = None


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

def get_balances_ordered(session: Session, offset: int = 0, limit: int = 100) -> list[Balance]:
    """Helper function to get balances ordered by date. Shared logic for consistency."""
    statement = select(Balance).order_by(desc(Balance.date)).offset(offset).limit(limit)
    return session.exec(statement).all()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()


# Balance
@app.get("/balance/", response_model=list[BalanceRead])
async def read_balance(
    session: SessionDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
) -> list[Balance]:
    return get_balances_ordered(session, offset, limit)
    

@app.post("/balance/", response_model=BalanceRead, status_code=status.HTTP_201_CREATED)
async def create_balance(balance: BalanceCreate, session: SessionDep):
    # Enforce unique date per balance entry
    # Validate YYYY-MM format
    try:
        validate_date_format(balance.date)
    except HTTPException:
        logger.warning(f"Attempted to create balance with invalid date={balance.date}")
        raise

    date_str = balance.date
    existing = session.exec(select(Balance).where(Balance.date == date_str)).first()
    if existing:
        logger.warning(f"Attempted to create balance for existing date={date_str}")
        raise HTTPException(status_code=409, detail="Balance for this date already exists")

    db_balance = Balance(date=date_str, balance=balance.balance)
    session.add(db_balance)
    session.commit()
    session.refresh(db_balance)
    logger.info(f"Created balance month={date_str} balance={balance.balance} id={db_balance.id}")
    return db_balance

@app.delete("/balance/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_balance(item_id: int, session: SessionDep):
    balance = session.get(Balance, item_id)
    if not balance:
        logger.warning(f"Attempted to delete non-existing balance id={item_id}")
        raise HTTPException(status_code=404, detail="Balance not found")
    logger.info(f"Deleted balance id={item_id}")
    session.delete(balance)
    session.commit()

# GET /metrics/delta?start=YYYY-MM&end=YYYY-MM (start and end required)
# Example response
# {
#   "range": { "from": "2025-01", "to": "2025-06" },
#   "items": [
#     { "month": "2025-01", "balance": 1000.0, "delta": null },
#     { "month": "2025-03", "balance": 1300.0, "delta": 300.0 },
#     { "month": "2025-06", "balance": 1250.0, "delta": -50.0 }
#   ],
#   "missing_months": ["2025-02", "2025-04", "2025-05"]
# }
@app.get("/metrics/delta/")
async def read_delta(
    session: SessionDep,
    start: str,
    end: str,
):
    # Validate date range
    validate_date_range(start, end)
    
    # Generate all months in range
    start_date = datetime.strptime(start, "%Y-%m")
    end_date = datetime.strptime(end, "%Y-%m")
    
    all_months = []
    current = start_date
    while current <= end_date:
        all_months.append(current.strftime("%Y-%m"))
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    # Get balances in range
    statement = select(Balance).where(Balance.date >= start, Balance.date <= end).order_by(asc(Balance.date))
    balances = session.exec(statement).all()
    
    # Create lookup for existing balances
    balance_dict = {b.date: b for b in balances}
    
    # Find missing months
    missing_months = [month for month in all_months if month not in balance_dict]
    
    # Build items with deltas
    items = []
    prev_balance = None
    
    for balance in balances:
        delta = None
        if prev_balance:
            delta = balance.balance - prev_balance.balance
        
        items.append({
            "month": balance.date,
            "balance": balance.balance,
            "delta": delta
        })
        prev_balance = balance
    
    return {
        "range": {"from": start, "to": end},
        "items": items,
        "missing_months": missing_months
    }

# GET /metrics/summary?start=YYYY-MM&end=YYYY-MM (start and end required)
# Response
# {
#   "range": { "from": "2025-01", "to": "2025-06" },
#   "start_balance": 1000.0,            // first available month in rage, or null
#   "end_balance": 1250.0,              // last available month in range, or null
#   "total_change": 250.0,              // end - start; null iff <2 data points
#   "avg_monthly_change": 125.0,        // average of all calculated deltas; null iff <2 data points
#   "last_month_delta": -50.0,          // last delta in the series; null iff <2 data points
#   "positive_months": 1,               // amount of deltas > 0
#   "negative_months": 1                // amount of deltas < 0
# }
@app.get("/metrics/summary/")
async def read_summary(
    session: SessionDep,
    start: str,
    end: str,
):
    # Validate date range
    validate_date_range(start, end)

    # Get balances in range directly from database
    statement = select(Balance).where(Balance.date >= start, Balance.date <= end).order_by(asc(Balance.date))
    balances = session.exec(statement).all()

    if not balances:
        return {
            "range": {"from": start, "to": end},
            "start_balance": None,
            "end_balance": None,
            "total_change": None,
            "avg_monthly_change": None,
            "last_month_delta": None,
            "positive_months": 0,
            "negative_months": 0
        }

    # Calculate basic metrics
    start_balance = balances[0].balance
    end_balance = balances[-1].balance
    total_change = end_balance - start_balance if len(balances) > 1 else None

    # Calculate true average monthly change based on time elapsed
    months_elapsed = calculate_months_between(start, end)
    avg_monthly_change = total_change / months_elapsed if months_elapsed > 0 and total_change is not None else None

    # Calculate consecutive month deltas for positive/negative counting
    consecutive_deltas = []
    last_month_delta = None

    for i in range(1, len(balances)):
        delta = balances[i].balance - balances[i-1].balance
        consecutive_deltas.append(delta)
        last_month_delta = delta  # This will be the last one

    # Count positive and negative consecutive changes
    positive_months = sum(1 for delta in consecutive_deltas if delta > 0)
    negative_months = sum(1 for delta in consecutive_deltas if delta < 0)

    return {
        "range": {"from": start, "to": end},
        "start_balance": start_balance,
        "end_balance": end_balance,
        "total_change": total_change,
        "avg_monthly_change": avg_monthly_change,
        "last_month_delta": last_month_delta,
        "positive_months": positive_months,
        "negative_months": negative_months
    }

# /metrics/current-month
# Response
# {
#   "month": "2025-09",
#   "balance": 1400.0,
#   "delta_vs_prev": 50.0   // null if no previous month
# }
@app.get("/metrics/current-month/")
async def read_current_month(session: SessionDep):
    # Get last balance from database
    statement = select(Balance).order_by(desc(Balance.date)).limit(1)
    last_balance = session.exec(statement).first()

    if not last_balance:
        return {
            "month": None,
            "balance": None,
            "delta_vs_prev": None
        }

    # Get previous balance from database
    prev_statement = select(Balance).where(Balance.date < last_balance.date).order_by(desc(Balance.date)).limit(1)
    prev_balance = session.exec(prev_statement).first()

    delta_vs_prev = last_balance.balance - prev_balance.balance if prev_balance else None

    return {
        "month": last_balance.date,
        "balance": last_balance.balance,
        "delta_vs_prev": delta_vs_prev
    }

class Message(SQLModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(SQLModel):
    messages: list[Message]
    stream: bool = False

class ChatResponse(SQLModel):
    role: Literal["assistant"]
    content: str

# request body { messages: Array<{role: "system"|"user"|"assistant", content: string}>, stream?: boolean }
@app.post("/ai/chat/", response_model=ChatResponse)
async def post_assistant(request: ChatRequest):



    logger.info(f"Received chat request: {request}")

    #Create a running input list we will add to over time
    input_messages = request.messages

    # Prompt the model with tools defined
    response = client.responses.create(
        prompt= {
            "id": "pmpt_68c1710ccc28819791f55143f28ac5ea0b21c468d0ed5fa7",
            "variables": { "date": datetime.now().strftime("%Y-%m-%d") },            
        },
        input=input_messages
    )   
    
    # Voeg modeloutput toe aan transcript
    input_messages += response.output

    # Verzamel tool-calls
    tool_calls = [o for o in response.output if o.type == "function_call"]

    if not tool_calls:
        # GEEN tool-calls -> het was een (clarifying) assistant-bericht.
        # -> Stuur dit meteen terug naar de frontend en STOP.
        print("NO TOOL CALLS")
        return ChatResponse(content=response.output_text, role="assistant")

    # Er ZIJN tool-calls -> voer ze uit
    print("TOOL CALLS")
    print(tool_calls)
    tasks = [
        call_function(tool_call.name, json.loads(tool_call.arguments))
        for tool_call in tool_calls
    ]
    results = await asyncio.gather(*tasks)
    for tool_call, result in zip(tool_calls, results):
        print("RESULT TOOL CALL")
        print(result)
        input_messages.append({
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": json.dumps(result)
        })

    # Nu pas de tweede model-call, omdat er nieuwe tool-data is
    response2 = client.responses.create(
        prompt={
            "id": "pmpt_68c1710ccc28819791f55143f28ac5ea0b21c468d0ed5fa7",
            "variables": { "date": datetime.now().strftime("%Y-%m-%d") },            
        },
        input=input_messages
    )   

    return ChatResponse(content=response2.output_text, role="assistant")
    


@app.get("/__test_500")
def force_500():
    raise RuntimeError("Boom! This is a test error")

def get_current_month_balance():
    logger.info(f"get_current_month_balance()")
    with Session(engine) as session:
        return  read_current_month(session)

def get_monthly_balance_deltas(start: str, end: str):
    logger.info(f"get_monthly_balance_deltas({start}, {end})")
    with Session(engine) as session:
        return read_delta(session, start, end)

def get_monthly_balance_summary(start: str, end: str):
    logger.info(f"get_monthly_balance_summary({start}, {end})")
    with Session(engine) as session:
        return read_summary(session, start, end)
    
def get_balance(month: str | None = None):
    logger.info(f"get_balance({month})")
    with Session(engine) as session:
        return read_summary(session, month, month)

async def call_function(name, args):
    logger.info(f"Calling function {name} with args {args}")

    if name == "get_balance":
        return await asyncio.to_thread(get_balance, args.get("month"))
    if name == "get_current_month_balance":
       return await asyncio.to_thread(get_current_month_balance)
    if name == "get_monthly_balance_deltas":
       return await asyncio.to_thread(get_monthly_balance_deltas, args.get("start"), args.get("end"))
    if name == "get_monthly_balance_summary":
        return await asyncio.to_thread(get_monthly_balance_summary, args.get("start"), args.get("end"))
        
     