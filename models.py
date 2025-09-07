# SQLAlchemy models for the consolidator function

from sqlalchemy import Column, String, JSON, ForeignKey, TIMESTAMP, func, Float
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import ARRAY

Base = declarative_base()

class Entity(Base):
    __tablename__ = "Entities"
    Eid = Column(String, primary_key=True)
    Type = Column(String)
    Properties = Column(JSON)
    Embedding = Column(ARRAY(Float))
    Communities = Column(ARRAY(String))

class Relationship(Base):
    __tablename__ = "Relationships"
    Rid = Column(String, primary_key=True)
    SourceEid = Column(String, ForeignKey("Entities.Eid"))
    TargetEid = Column(String, ForeignKey("Entities.Eid"))
    Type = Column(String)
    Properties = Column(JSON)

class InstanceOf(Base):
    __tablename__ = "InstanceOf"
    InstanceEid = Column(String, ForeignKey("Entities.Eid"), primary_key=True)
    ClassEid = Column(String, ForeignKey("Entities.Eid"), primary_key=True)

class WorkflowStatus(Base):
    __tablename__ = "WorkflowStatus"
    BatchId = Column(String, primary_key=True)
    Status = Column(String)
    UpdatedAt = Column(TIMESTAMP, server_default=func.now())

class Community(Base):
    __tablename__ = "Communities"
    CommunityId = Column(String, primary_key=True)
    Summary = Column(String)
    Embedding = Column(ARRAY(Float))