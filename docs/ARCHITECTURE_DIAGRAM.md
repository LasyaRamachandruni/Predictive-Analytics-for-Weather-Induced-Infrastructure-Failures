# Architecture Diagram
## Predictive Analytics for Weather-Induced Infrastructure Failures

This document contains visual architecture diagrams in Mermaid format that can be rendered in Markdown viewers.

---

## System Architecture Overview

```mermaid
graph TB
    subgraph "Data Ingestion Layer"
        A[Configuration File] --> B{Mode Selection}
        B -->|Demo| C[Synthetic Data Generator]
        B -->|Real| D[NOAA Data Downloader]
        C --> E[Raw DataFrame]
        D --> E
    end
    
    subgraph "Feature Engineering Layer"
        E --> F[Time Series Alignment]
        F --> G[Lag Features]
        G --> H[Rolling Statistics]
        H --> I[Target Transformations]
        I --> J[Engineered DataFrame]
    end
    
    subgraph "Data Preparation Layer"
        J --> K[Chronological Split]
        K --> L[Tabular Format]
        K --> M[Sequence Format]
        L --> N[Feature Scaling]
        M --> N
    end
    
    subgraph "Model Training Layer"
        N --> O[LSTM Model]
        N --> P[Random Forest]
        N --> Q[XGBoost]
        P --> R[Tabular Blend]
        Q --> R
        O --> S[Hybrid Ensemble]
        R --> S
    end
    
    subgraph "Evaluation Layer"
        S --> T[Metrics Computation]
        S --> U[Visualization]
        T --> V[Artifacts]
        U --> V
    end
    
    style A fill:#e1f5ff
    style E fill:#fff4e1
    style J fill:#e8f5e9
    style S fill:#fce4ec
    style V fill:#f3e5f5
```

---

## Data Flow Diagram

```mermaid
flowchart LR
    subgraph "Input Sources"
        A1[Demo Mode<br/>Synthetic Data]
        A2[Real Mode<br/>NOAA/GHCN]
    end
    
    A1 --> B[Raw DataFrame]
    A2 --> B
    
    B --> C[Feature Engineering]
    C --> D[Engineered DataFrame<br/>~150 features]
    
    D --> E1[Tabular Format<br/>N × features]
    D --> E2[Sequence Format<br/>N × 24 × features]
    
    E1 --> F1[Random Forest]
    E1 --> F2[XGBoost]
    F1 --> G[Tabular Blend]
    F2 --> G
    
    E2 --> F3[LSTM]
    
    G --> H[Hybrid Ensemble]
    F3 --> H
    
    H --> I[Predictions]
    H --> J[Metrics]
    H --> K[Visualizations]
    
    style B fill:#fff4e1
    style D fill:#e8f5e9
    style H fill:#fce4ec
    style I fill:#f3e5f5
```

---

## Feature Engineering Pipeline

```mermaid
graph TD
    A[Raw DataFrame] --> B[Time Series Alignment]
    B --> C[Group by Region]
    C --> D[Create Continuous Timeline]
    D --> E[Forward Fill Missing Values]
    
    E --> F[Lag Features]
    F --> F1[Lag 1, 3, 6, 12, 24]
    
    E --> G[Rolling Statistics]
    G --> G1[Rolling Mean 3, 6, 12, 24]
    G --> G2[Rolling Max 3, 6, 12, 24]
    G --> G3[Rolling Std 3, 6, 12, 24]
    
    E --> H[Target Transformations]
    H --> H1[Difference]
    H --> H2[Percentage Change]
    
    F1 --> I[Feature Collection]
    G1 --> I
    G2 --> I
    G3 --> I
    H1 --> I
    H2 --> I
    
    I --> J[Fill NaN with 0]
    J --> K[Drop Essential NaN Rows]
    K --> L[Engineered DataFrame]
    
    style A fill:#fff4e1
    style L fill:#e8f5e9
```

---

## Model Architecture

```mermaid
graph TB
    subgraph "LSTM Model"
        A1[Input Sequence<br/>batch × 24 × 151] --> A2[LSTM Layer 1<br/>128 hidden units]
        A2 --> A3[LSTM Layer 2<br/>128 hidden units]
        A3 --> A4[Dropout 0.2]
        A4 --> A5[Fully Connected]
        A5 --> A6[Output<br/>batch × 1]
    end
    
    subgraph "Tabular Models"
        B1[Input Features<br/>batch × 151] --> B2[Random Forest<br/>400 trees]
        B1 --> B3[XGBoost<br/>500 trees]
        B2 --> B4[RF Prediction]
        B3 --> B5[XGB Prediction]
        B4 --> B6[Blended Tabular<br/>0.5 × RF + 0.5 × XGB]
        B5 --> B6
    end
    
    A6 --> C[Hybrid Ensemble]
    B6 --> C
    C --> D[Final Prediction<br/>0.5 × LSTM + 0.5 × Tabular]
    
    style A1 fill:#e1f5ff
    style B1 fill:#fff4e1
    style C fill:#fce4ec
    style D fill:#c8e6c9
```

---

## Training Process Flow

```mermaid
sequenceDiagram
    participant Config
    participant Pipeline
    participant LSTM
    participant RF
    participant XGB
    participant Ensemble
    participant Eval
    
    Config->>Pipeline: Load Configuration
    Pipeline->>Pipeline: Data Ingestion
    Pipeline->>Pipeline: Feature Engineering
    Pipeline->>Pipeline: Data Splitting
    Pipeline->>Pipeline: Format Creation
    
    Pipeline->>LSTM: Train on Sequences
    LSTM->>LSTM: Forward/Backward Pass
    LSTM->>LSTM: Early Stopping Check
    LSTM-->>Pipeline: Trained Model
    
    Pipeline->>RF: Train on Tabular
    RF-->>Pipeline: Trained Model
    
    Pipeline->>XGB: Train on Tabular
    XGB-->>Pipeline: Trained Model
    
    Pipeline->>Ensemble: Create Hybrid
    Ensemble->>Ensemble: Blend Predictions
    Ensemble-->>Eval: Final Predictions
    
    Eval->>Eval: Compute Metrics
    Eval->>Eval: Generate Plots
    Eval->>Eval: Save Artifacts
```

---

## Data Split Visualization

```mermaid
gantt
    title Chronological Data Split
    dateFormat YYYY-MM-DD
    section Train
    Training Data (70%)    :2024-01-01, 70d
    section Validation
    Validation Data (15%)  :2024-03-11, 15d
    section Test
    Test Data (15%)        :2024-03-26, 15d
```

---

## Ensemble Blending Strategy

```mermaid
graph LR
    A[LSTM Prediction<br/>Temporal Patterns] --> C[Hybrid Prediction]
    B[Tabular Blend<br/>Feature Interactions] --> C
    
    B1[Random Forest<br/>Feature Interactions] --> B
    B2[XGBoost<br/>Gradient Boosting] --> B
    
    C --> D[Final Output]
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#fce4ec
    style D fill:#c8e6c9
```

---

## Component Interaction Diagram

```mermaid
graph TB
    subgraph "Data Components"
        DC1[download_and_preprocess.py]
        DC2[data_pipeline.py]
    end
    
    subgraph "Model Components"
        MC1[lstm_model.py]
        MC2[ensemble.py]
        MC3[train_ensemble.py]
    end
    
    subgraph "Utility Components"
        UC1[io.py]
        UC2[metrics.py]
    end
    
    subgraph "Visualization Components"
        VC1[plot_results.py]
        VC2[map_failures.py]
    end
    
    DC1 --> DC2
    DC2 --> MC3
    MC3 --> MC1
    MC3 --> MC2
    MC3 --> UC1
    MC3 --> UC2
    MC3 --> VC1
    VC2 --> UC1
    
    style DC1 fill:#e1f5ff
    style DC2 fill:#e1f5ff
    style MC1 fill:#fff4e1
    style MC2 fill:#fff4e1
    style MC3 fill:#fff4e1
    style UC1 fill:#e8f5e9
    style UC2 fill:#e8f5e9
    style VC1 fill:#fce4ec
    style VC2 fill:#fce4ec
```

---

## Feature Engineering Details

```mermaid
mindmap
  root((Feature Engineering))
    Time Alignment
      Continuous Timeline
      Forward Fill
      Group by Region
    Lag Features
      Lag 1
      Lag 3
      Lag 6
      Lag 12
      Lag 24
    Rolling Statistics
      Mean
      Max
      Std
      Windows: 3, 6, 12, 24
    Target Transformations
      Difference
      Percentage Change
```

---

## Model Comparison

```mermaid
graph TB
    subgraph "Input Formats"
        IF1[Tabular<br/>N × Features]
        IF2[Sequences<br/>N × 24 × Features]
    end
    
    subgraph "Models"
        M1[Random Forest<br/>Strengths:<br/>• Feature interactions<br/>• Robust to outliers]
        M2[XGBoost<br/>Strengths:<br/>• High accuracy<br/>• Feature importance]
        M3[LSTM<br/>Strengths:<br/>• Temporal patterns<br/>• Long dependencies]
    end
    
    subgraph "Outputs"
        O1[Tabular Blend<br/>RF + XGB]
        O2[Hybrid Ensemble<br/>Tabular + LSTM]
    end
    
    IF1 --> M1
    IF1 --> M2
    IF2 --> M3
    
    M1 --> O1
    M2 --> O1
    O1 --> O2
    M3 --> O2
    
    style IF1 fill:#e1f5ff
    style IF2 fill:#e1f5ff
    style O2 fill:#c8e6c9
```

---

## Deployment Architecture

```mermaid
graph TB
    subgraph "Training Phase"
        T1[Training Data] --> T2[Model Training]
        T2 --> T3[Model Artifacts]
    end
    
    subgraph "Inference Phase"
        I1[New Weather Data] --> I2[Feature Engineering]
        I2 --> I3[Load Models]
        T3 --> I3
        I3 --> I4[Generate Predictions]
        I4 --> I5[Risk Assessment]
    end
    
    style T2 fill:#fff4e1
    style I4 fill:#fce4ec
    style I5 fill:#c8e6c9
```

---

These diagrams provide visual representations of the system architecture, data flow, and component interactions. They can be rendered in Markdown viewers that support Mermaid (like GitHub, GitLab, or VS Code with Mermaid extension).

