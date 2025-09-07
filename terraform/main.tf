resource "google_cloud_run_v2_service" "consolidator" {
  project  = var.project_id
  name     = "consolidator-fn"
  location = var.location
  deletion_protection = false
  ingress = "INGRESS_TRAFFIC_INTERNAL_ONLY"

  template {
    service_account = var.consolidator_sa_email
    timeout         = "1800s" # 30 minutes for potentially long-running consolidations
    scaling {
      min_instance_count = 0 # Can scale to zero
      max_instance_count = 1
    }
    max_instance_request_concurrency = 1
    vpc_access {
      connector = var.vpc_connector
      egress = "ALL_TRAFFIC"
    }
    containers {
        image = "${var.image_url}:${var.image_tag}"
        resources {
          limits = {
            "memory": "2Gi",
            "cpu": "1"
          }
        }
       
        
        
        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }
        env {
          name  = "REDIS_HOST"
          value = var.redis_host
        }
        env {
          name  = "REDIS_PORT"
          value = var.redis_port
        }
        
        env {
          name  = "MEMGRAPH_HOST"
          value = var.memgraph_host
        }
        env {
          name  = "MEMGRAPH_PORT"
          value = var.memgraph_port
        }
        env {
          name  = "MEMGRAPH_USER"
          value = var.memgraph_user
        }
        env {
          name  = "MEMGRAPH_PASSWORD"
          value = var.memgraph_password
        }
        env {
          name  = "NEO4J_USERNAME"
          value = var.memgraph_user
        }
        env {
          name  = "NEO4J_PASSWORD"
          value = var.memgraph_password
        }
        env {
          name  = "SPANNER_INSTANCE_ID"
          value = var.spanner_instance_id
        }
        env {
          name  = "SPANNER_DATABASE_ID"
          value = var.spanner_database_id
        }
        env {
          name  = "LOCATION"
          value = var.region
        }
        env {
          name  = "USE_GEMINI_EMBEDDINGS"
          value = var.use_gemini_embeddings 
        }
        env {
          name = "GEMINI_API_KEY_SECRET_ID"
          value = var.gemini_api_key_secret_id
        }
        env {
          name  = "EMBEDDING_SERVICE_URL"
          value = data.google_cloud_run_v2_service.embedding_service.uri
        }
      }
  }
  depends_on = [
    data.google_cloud_run_v2_service.embedding_service
  ]


}

resource "google_project_iam_member" "consolidator_sa_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${var.consolidator_sa_email}"
}

resource "google_cloud_run_v2_service_iam_member" "embedding_invoker" {
  project  = var.project_id
  name     = "graphrag-embedding"
  location = var.location
  role     = "roles/run.invoker"
  member   = "serviceAccount:${var.consolidator_sa_email}"
}

resource "google_eventarc_trigger" "consolidator_trigger" {
  project  = var.project_id
  name     = "consolidator-trigger-pubsub"
  location = var.location

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.pubsub.topic.v1.messagePublished"
  }

  destination {
    cloud_run_service {
      service = google_cloud_run_v2_service.consolidator.name
      region  = var.location
    }
  }

  transport {
    pubsub {
      topic = data.google_pubsub_topic.consolidator.id
    }
  }

  service_account = var.consolidator_sa_email
}


data "google_pubsub_topic" "consolidator" {
  name = var.topic_name
  project  = var.project_id
}

data "google_cloud_run_v2_service" "embedding_service" {
  name     = "graphrag-embedding"
  location = var.location
}