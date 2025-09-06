CREATE TABLE Entities (
  Eid STRING(MAX) NOT NULL,
  Type STRING(MAX) NOT NULL,
  Properties JSON,
  Embedding ARRAY<FLOAT64>(vector_length=>768),
  Communities ARRAY<STRING(MAX)>,
) PRIMARY KEY (Eid);

CREATE TABLE Relationships (
  Rid STRING(MAX) NOT NULL,
  SourceEid STRING(MAX) NOT NULL,
  TargetEid STRING(MAX) NOT NULL,
  Type STRING(MAX) NOT NULL,
  Properties JSON,
  CONSTRAINT FK_Source FOREIGN KEY (SourceEid) REFERENCES Entities (Eid),
  CONSTRAINT FK_Target FOREIGN KEY (TargetEid) REFERENCES Entities (Eid),
) PRIMARY KEY (Rid);

CREATE TABLE InstanceOf (
  InstanceEid STRING(MAX) NOT NULL,
  ClassEid STRING(MAX) NOT NULL,
  CONSTRAINT FK_Instance FOREIGN KEY (InstanceEid) REFERENCES Entities (Eid),
  CONSTRAINT FK_Class FOREIGN KEY (ClassEid) REFERENCES Entities (Eid),
) PRIMARY KEY (InstanceEid, ClassEid);

CREATE TABLE ProcessedDocuments (
  BatchId STRING(MAX) NOT NULL,
  DocumentId STRING(MAX) NOT NULL,
  ProcessedAt TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
) PRIMARY KEY (BatchId, DocumentId);

CREATE TABLE Communities (
  CommunityId STRING(MAX) NOT NULL,
  Summary STRING(MAX),
  Embedding ARRAY<FLOAT64>(vector_length=>768),
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

CREATE VECTOR INDEX EntitiesEmbeddingIndex ON Entities(Embedding) WHERE Embedding IS NOT NULL OPTIONS(distance_type = 'COSINE');

CREATE VECTOR INDEX CommunitiesEmbeddingIndex ON Communities(Embedding) WHERE Embedding IS NOT NULL OPTIONS(distance_type = 'COSINE');