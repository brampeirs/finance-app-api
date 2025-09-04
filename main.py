from typing import Annotated
import re

from fastapi import FastAPI, Depends, Query, HTTPException, status

from fastapi.middleware.cors import CORSMiddleware

from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import asc  

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tijdelijk: alles toestaan
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
    statement = select(Balance).order_by(asc(Balance.date)).offset(offset).limit(limit)
    return session.exec(statement).all()
    

@app.post("/balance/", response_model=BalanceRead, status_code=status.HTTP_201_CREATED)
async def create_balance(balance: BalanceCreate, session: SessionDep):
    # Enforce unique date per balance entry
    # Validate YYYY-MM format
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", balance.date):
        raise HTTPException(status_code=422, detail="date must be in 'YYYY-MM' format")

    date_str = balance.date
    existing = session.exec(select(Balance).where(Balance.date == date_str)).first()
    if existing:
        raise HTTPException(status_code=409, detail="Balance for this date already exists")

    db_balance = Balance(date=date_str, balance=balance.balance)
    session.add(db_balance)
    session.commit()
    session.refresh(db_balance)
    return db_balance

@app.delete("/balance/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_balance(item_id: int, session: SessionDep):
    balance = session.get(Balance, item_id)
    if not balance:
        raise HTTPException(status_code=404, detail="Balance not found")
    session.delete(balance)
    session.commit()

