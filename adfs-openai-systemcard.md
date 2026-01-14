```mermaid
flowchart LR
  User[User];
  Browser[Browser];
  App[Enterprise App];
  ADFS[ADFS IdP];
  AD[Active Directory];
  Agent[AI Agent];
  Tools[Agent Tools];
  Resource[Enterprise Resources];

  User --> Browser;
  Browser --> App;
  App -->|Auth Request| ADFS;
  ADFS -->|Authenticate| AD;
  ADFS -->|Claims Token| App;
  App -->|Context| Agent;
  Agent --> Tools;
  Tools --> Resource;
