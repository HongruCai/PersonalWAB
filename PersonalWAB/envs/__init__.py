
def get_env(env_name: str, user_mode: str, user_model: str, task_split: str, max_steps: int):
    if env_name == 'pwab':
        from PersonalWAB.envs.pwab import MockPWADomainEnv

        return MockPWADomainEnv(
            user_mode=user_mode, user_model=user_model, task_split=task_split, max_steps=max_steps
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")
