# 🔐 ADFS / SSO / AI Agent 연계 아키텍처 (One Page)

본 문서는 **엔터프라이즈 환경에서 ADFS 기반 SSO와 AI Agent(ChatGPT Agent 등)를
연계하는 기본 아키텍처 구조**를  
**GitHub 다크모드 기준 가독성 최적화 + Mermaid 다이어그램**으로
한 페이지에 정리한 자료입니다.

---

## 1. 전체 연계 구조 개요

```mermaid
flowchart LR
    User[👤 User]
    Browser[🌐 Browser]
    App[🏢 Enterprise App\n(Web Portal)]
    ADFS[🔐 ADFS\n(IdP)]
    AD[🗂️ Active Directory]
    Agent[🤖 AI Agent]
    Tools[🧰 Agent Tools\n(Browser / API / Terminal)]
    Resource[📁 Enterprise Resources\n(Docs / Systems)]

    User --> Browser
    Browser --> App
    App -->|Auth Request\nSAML / OIDC| ADFS
    ADFS -->|LDAP / Kerberos| AD
    ADFS -->|Claims Token| App
    App -->|Context| Agent
    Agent --> Tools
    Tools --> Resource

sequenceDiagram
    participant U as 👤 User
    participant B as 🌐 Browser
    participant A as 🏢 App
    participant F as 🔐 ADFS
    participant D as 🗂️ AD

    U->>B: Access App
    B->>A: Request Resource
    A->>F: Authn Request\n(SAML / OIDC)
    F->>D: Authenticate User
    D-->>F: Auth Result
    F-->>A: Claims Token
    A-->>B: SSO Session Established

flowchart TD
    App[🏢 Authenticated App]
    Claims[🪪 User Claims\n(Role / Group)]
    Agent[🤖 AI Agent]
    Policy[📜 Security Policy]
    Action[⚙️ Agent Action]
    Block[⛔ Block or Ask User]

    App --> Claims
    Claims --> Agent
    Agent --> Policy
    Policy -->|Allow| Action
    Policy -->|Deny| Block

flowchart LR
    Identity[🔐 ADFS Identity]
    Identity --> Least[🔒 Least Privilege]
    Identity --> Audit[📊 Audit Log]
    Identity --> Confirm[✅ User Confirmation]

flowchart LR
    ADFS[🔐 ADFS]
    App[🏢 App]
    Agent[🤖 AI Agent]
    Systems[🏭 Enterprise Systems]

    ADFS -->|SSO / Claims| App
    App -->|Context| Agent
    Agent -->|Controlled Action| Systems
```
