import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalCNN(nn.Module):
    def __init__(self, num_sex_categories, num_loc_categories):
        super().__init__()

        # ----------------------------
        # IMAGE ENCODER (ResNet18)
        # ----------------------------
        self.cnn = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn.fc = nn.Identity()  # output = (batch, 512)

        image_feature_dim = 512

        # ----------------------------
        # METADATA ENCODERS
        # ----------------------------

        # Sex: categorical embedding (e.g., male/female/unknown)
        self.sex_emb_dim = 4
        self.sex_emb = nn.Embedding(
            num_embeddings=num_sex_categories,
            embedding_dim=self.sex_emb_dim
        )

        # Anatomical location: embedding (more categories)
        self.loc_emb_dim = 8
        self.loc_emb = nn.Embedding(
            num_embeddings=num_loc_categories,
            embedding_dim=self.loc_emb_dim
        )

        # Age: scalar → small dense projection
        self.age_dim = 8
        self.age_fc = nn.Linear(1, self.age_dim)

        metadata_dim = self.sex_emb_dim + self.loc_emb_dim + self.age_dim

        # ----------------------------
        # FUSION + CLASSIFIER
        # ----------------------------
        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + metadata_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, images, age, sex, loc):
        # Image features
        img_features = self.cnn(images)  # (batch, 512)

        # Metadata embeddings
        sex_emb  = self.sex_emb(sex)            # (batch, sex_emb_dim)
        loc_emb  = self.loc_emb(loc)            # (batch, loc_emb_dim)
        age_proj = self.age_fc(age)             # (batch, age_dim)

        meta_vec = torch.cat([sex_emb, loc_emb, age_proj], dim=1)

        # Fusion
        fused = torch.cat([img_features, meta_vec], dim=1)

        logits = self.classifier(fused)
        return logits.squeeze(1)
