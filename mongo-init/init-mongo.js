db = db.getSiblingDB("arti");

// Check and create collections if they do not exist
if (!db.getCollectionNames().includes("lost_paintings")) {
  db.createCollection("lost_paintings");
}

if (!db.getCollectionNames().includes("lost_paintings_training")) {
  db.createCollection("lost_paintings_training");
}

if (!db.getCollectionNames().includes("paintings")) {
  db.createCollection("paintings");
}