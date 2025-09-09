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