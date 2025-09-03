from typing import Union, Annotated

from fastapi import FastAPI, Depends, Query

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
@app.get("/balance/")
async def read_balance(session: SessionDep, offset: int = 0, limit: Annotated[int, Query(le=100)] = 100) -> list[Balance]:
    statement = select(Balance).order_by(asc(Balance.date)).offset(offset).limit(limit)
    return session.exec(statement).all()
    


@app.post("/balance/")
async def create_balance(balance: Balance, session: SessionDep):
    session.add(balance)
    session.commit()
    session.refresh(balance)
    return balance
