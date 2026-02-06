# syntax=docker/dockerfile:1.0
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.0

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND="noninteractive"
ENV LC_ALL="C"
ENV TZ="UTC"

# --- 共通パッケージとツールのインストール ---
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        git \
        gosu \
        htop \
        nvtop \
        wget \
        g++ \
        vim \
        sudo \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Condaのセットアップ (共通) ---
ENV CONDAHOME="/opt/conda"
ENV PATH="${CONDAHOME}/bin:${PATH}"
RUN wget -q -P /tmp https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh

COPY environment.yml /opt/conda/environment.yml
RUN conda install -y python=3.12 \
    && conda env update -n base -f /opt/conda/environment.yml \
    && conda clean -y --all --force-pkgs-dirs

# --- ユーザーとEntrypointの設定 (統合部分) ---
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# ユーザー作成と権限設定
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    # sudo権限の付与
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # Condaの所有権変更
    && chown -R $USERNAME:$USERNAME /opt/conda

# pip --user 用のパス設定
ENV PATH="/home/${USERNAME}/.local/bin:$PATH"

# Entrypointスクリプトの作成
COPY --chmod=755 <<EOF /usr/local/bin/entrypoint.sh
#!/bin/bash
set -e

# 環境変数 UID/GID が渡された場合、ユーザーIDを更新する
TARGET_UID=\${UID:-$USER_UID}
TARGET_GID=\${GID:-$USER_GID}
TARGET_USER=$USERNAME

# 現在のIDと異なる場合のみ更新処理を行う
if [ "\$(id -u \$TARGET_USER)" != "\$TARGET_UID" ] || [ "\$(id -g \$TARGET_USER)" != "\$TARGET_GID" ]; then
    groupmod -g \$TARGET_GID -o \$TARGET_USER
    usermod -u \$TARGET_UID -g \$TARGET_GID -o \$TARGET_USER
    
    # 必要に応じて所有権を修正 (ホームディレクトリなど)
    chown -R \$TARGET_UID:\$TARGET_GID /home/\$TARGET_USER
fi

# コマンド実行 (gosuを使ってユーザー権限で実行)
exec gosu \$TARGET_USER "\$@"
EOF

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]