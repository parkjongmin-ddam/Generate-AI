# ADFS / SSO / AI Agent 연계 아키텍처 (One Page)

본 문서는 **엔터프라이즈 환경에서 ADFS 기반 SSO와 AI Agent(ChatGPT Agent 등)를
연계하는 기본 아키텍처 구조**를 Mermaid 다이어그램으로 간략히 정리한 자료입니다.

---

## 1. 전체 연계 구조 개요

```mermaid
flowchart LR
    User[User / Employee]
    Browser[Browser]
    App[Enterprise App<br/>(Web / Portal)]
    ADFS[ADFS<br/>IdP]
    AD[Active Directory]
    Agent[AI Agent]
    Tools[Agent Tools<br/>(Browser / API / Terminal)]
    Resource[Enterprise Resources<br/>(Docs / Systems)]

    User --> Browser
    Browser --> App
    App -->|Auth Request<br/>(SAML / OIDC)| ADFS
    ADFS -->|LDAP / Kerberos| AD
    ADFS -->|Claims Token| App
    App --> Agent
    Agent --> Tools
    Tools --> Resource

sequenceDiagram
    participant U as User
    participant B as Browser
    participant A as App
    participant F as ADFS
    participant D as Active Directory

    U->>B: App Access
    B->>A: Request Resource
    A->>F: Authn Request (SAML / OIDC)
    F->>D: User Authentication
    D-->>F: Auth Result
    F-->>A: Claims Token
    A-->>B: SSO Session Established

flowchart TD
    App[Authenticated App]
    Token[User Identity / Claims]
    Agent[AI Agent]
    Policy[Security Policy]
    Action[Agent Action]

    App -->|Context + Claims| Agent
    Agent --> Policy
    Policy -->|Allow| Action
    Policy -->|Deny| Block[Blocked / Ask User]

flowchart LR
    Identity[ADFS Identity]
    Identity --> LeastPrivilege[Least Privilege]
    Identity --> Audit[Audit / Logging]
    Identity --> Confirmation[User Confirmation]

flowchart LR
    ADFS -->|SSO / Claims| App
    App -->|Context| Agent
    Agent -->|Controlled Action| Enterprise[Enterprise Systems]
