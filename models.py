# SQLAlchemy models for the consolidator function

from sqlalchemy import Column, String, JSON, ForeignKey, TIMESTAMP, func, Float
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import ARRAY
from sqlalchemy.ext.compiler import compiles

Base = declarative_base()

# Custom SQLAlchemy type for Spanner Vector Arrays
class SpannerVectorArray(ARRAY):
    """
    A custom ARRAY type for Google Cloud Spanner that includes
    the vector_length annotation for DDL generation.
    """
    def __init__(self, item_type, dimensions=None, zero_indexes=False, vector_length=None):
        super().__init__(item_type, dimensions=dimensions, zero_indexes=zero_indexes)
        if not isinstance(item_type, type(Float)):
            raise ValueError("SpannerVectorArray only supports Float item_type.")
        if not isinstance(vector_length, int) or vector_length <= 0:
            raise ValueError("vector_length must be a positive integer.")
        self.vector_length = vector_length

@compiles(SpannerVectorArray, "spanner")
def compile_spanner_vector_array(element, compiler, **kw):
    """
    Custom DDL compilation for SpannerVectorArray to include vector_length.
    Generates ARRAY<FLOAT64>(vector_length=>X)
    """
    item_type_sql = compiler.process(element.item_type, **kw)
    return f"ARRAY<{item_type_sql}>(vector_length=>{element.vector_length})"

class Entity(Base):
    __tablename__ = "Entities"
    Eid = Column(String, primary_key=True)
    Type = Column(String)
    Properties = Column(JSON)
    ClusteringEmbedding = Column(SpannerVectorArray(Float, vector_length=768))
    RetrievalDocumentEmbedding = Column(SpannerVectorArray(Float, vector_length=768))
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
    ClusteringEmbedding = Column(SpannerVectorArray(Float, vector_length=768))
    RetrievalDocumentEmbedding = Column(SpannerVectorArray(Float, vector_length=768))

class EntityCommunity(Base):
    __tablename__ = "EntityCommunity"
    Eid = Column(String, ForeignKey("Entities.Eid"), primary_key=True)
    CommunityId = Column(String, ForeignKey("Communities.CommunityId"), primary_key=True)