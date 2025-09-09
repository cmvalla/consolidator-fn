CREATE TABLE Entities (
  Eid STRING(MAX) NOT NULL,
  Type STRING(MAX) NOT NULL,
  Properties JSON,
  ClusteringEmbedding ARRAY<FLOAT64>(vector_length=>768),
  RetrievalDocumentEmbedding ARRAY<FLOAT64>(vector_length=>768),
  Communities ARRAY<STRING(MAX)>,
) PRIMARY KEY (Eid);

CREATE TABLE Relationships (
  Rid STRING(MAX) NOT NULL,
  SourceEid STRING(MAX) NOT NULL,
  TargetEid STRING(MAX) NOT NULL,
  Type STRING(MAX) NOT NULL,
  Properties JSON
) PRIMARY KEY (Rid),
INTERLEAVE IN PARENT Entities ON DELETE CASCADE;

CREATE TABLE InstanceOf (
  InstanceEid STRING(MAX) NOT NULL,
  ClassEid STRING(MAX) NOT NULL
) PRIMARY KEY (InstanceEid, ClassEid),
INTERLEAVE IN PARENT Entities ON DELETE CASCADE;

CREATE TABLE ProcessedDocuments (
  BatchId STRING(MAX) NOT NULL,
  DocumentId STRING(MAX) NOT NULL,
  ProcessedAt TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
) PRIMARY KEY (BatchId, DocumentId);

CREATE TABLE Communities (
  CommunityId STRING(MAX) NOT NULL,
  Summary STRING(MAX),
  ClusteringEmbedding ARRAY<FLOAT64>(vector_length=>768),
  RetrievalDocumentEmbedding ARRAY<FLOAT64>(vector_length=>768),
) PRIMARY KEY (CommunityId);

CREATE TABLE EntityCommunity (
  Eid STRING(MAX) NOT NULL,
  CommunityId STRING(MAX) NOT NULL,
  CONSTRAINT FK_Entity FOREIGN KEY (Eid) REFERENCES Entities (Eid),
  CONSTRAINT FK_Community FOREIGN KEY (CommunityId) REFERENCES Communities (CommunityId),
) PRIMARY KEY (Eid, CommunityId);

CREATE PROPERTY GRAPH my_graph
    NODE TABLES (
        Entities,
        Communities
    )
    EDGE TABLES (
        Relationships
            SOURCE KEY (SourceEid) REFERENCES Entities (Eid)
            DESTINATION KEY (TargetEid) REFERENCES Entities (Eid),
        InstanceOf
            SOURCE KEY (InstanceEid) REFERENCES Entities (Eid)
            DESTINATION KEY (ClassEid) REFERENCES Entities (Eid),
        EntityCommunity
            SOURCE KEY (Eid) REFERENCES Entities (Eid)
            DESTINATION KEY (CommunityId) REFERENCES Communities (CommunityId)
    );

CREATE VECTOR INDEX EntitiesClusteringEmbeddingIndex ON Entities(ClusteringEmbedding) WHERE ClusteringEmbedding IS NOT NULL OPTIONS(distance_type = 'COSINE');
CREATE VECTOR INDEX EntitiesRetrievalDocumentEmbeddingIndex ON Entities(RetrievalDocumentEmbedding) WHERE RetrievalDocumentEmbedding IS NOT NULL OPTIONS(distance_type = 'COSINE');

CREATE VECTOR INDEX CommunitiesClusteringEmbeddingIndex ON Communities(ClusteringEmbedding) WHERE ClusteringEmbedding IS NOT NULL OPTIONS(distance_type = 'COSINE');
CREATE VECTOR INDEX CommunitiesRetrievalDocumentEmbeddingIndex ON Communities(RetrievalDocumentEmbedding) WHERE RetrievalDocumentEmbedding IS NOT NULL OPTIONS(distance_type = 'COSINE');