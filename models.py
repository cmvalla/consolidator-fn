# SQLAlchemy models for the consolidator function

from sqlalchemy import Column, String, JSON, ForeignKey, TIMESTAMP, func, Float
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import ARRAY, TypeDecorator
from sqlalchemy.ext.compiler import compiles

Base = declarative_base()

# Custom SQLAlchemy type for Spanner Vector Arrays
class SpannerVector(TypeDecorator):
    """
    A custom SQLAlchemy type for Spanner's ARRAY<FLOAT64>(vector_length=>...)
    """
    impl = ARRAY(Float)  # Base implementation is a SQLAlchemy ARRAY of Floats
    cache_ok = True      # Mark as cacheable for performance

    def __init__(self, dimensions: int, **kw):
        super().__init__(**kw)
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self.dimensions = dimensions

    def process_bind_param(self, value, dialect):
        if value is not None and len(value) != self.dimensions:
            raise ValueError(f"Vector length must be {self.dimensions}, but got {len(value)}")
        return value

    def process_result_value(self, value, dialect):
        return value

@compiles(SpannerVector, 'spanner')
def compile_spanner_vector(element, compiler, **kw):
    """
    Custom compilation for SpannerVector type specifically for the 'spanner' dialect.
    This generates the DDL: ARRAY<FLOAT64>(vector_length=>X)
    """
    float_type = "FLOAT64"
    return f"ARRAY<{float_type}>(vector_length=>{element.dimensions})"

class Entity(Base):
    __tablename__ = "Entities"
    Eid = Column(String, primary_key=True)
    Type = Column(String)
    Properties = Column(JSON)
    ClusteringEmbedding = Column(SpannerVector(dimensions=768))
    RetrievalDocumentEmbedding = Column(SpannerVector(dimensions=768))
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
    ClusteringEmbedding = Column(SpannerVector(dimensions=768))
    RetrievalDocumentEmbedding = Column(SpannerVector(dimensions=768))

class EntityCommunity(Base):
    __tablename__ = "EntityCommunity"
    Eid = Column(String, ForeignKey("Entities.Eid"), primary_key=True)
    CommunityId = Column(String, ForeignKey("Communities.CommunityId"), primary_key=True)