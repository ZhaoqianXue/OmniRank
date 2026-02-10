"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

const GOOGLE_SCRIPT_ID = "omnirank-google-gsi-script";
const GOOGLE_SCRIPT_SRC = "https://accounts.google.com/gsi/client";
const GOOGLE_AUTH_STORAGE_KEY = "omnirank_google_auth_v1";

let googleScriptLoadPromise: Promise<void> | null = null;

interface GoogleTokenResponse {
  access_token?: string;
  error?: string;
  error_description?: string;
}

interface GoogleSdkError {
  type?: string;
}

interface GoogleTokenClient {
  requestAccessToken: (options?: { prompt?: string }) => void;
}

interface GoogleTokenClientConfig {
  client_id: string;
  scope: string;
  callback: (response: GoogleTokenResponse) => void;
  error_callback?: (error: GoogleSdkError) => void;
}

interface GoogleOAuth2Namespace {
  initTokenClient: (config: GoogleTokenClientConfig) => GoogleTokenClient;
  revoke: (token: string, callback?: () => void) => void;
}

interface GoogleAccountsNamespace {
  oauth2?: GoogleOAuth2Namespace;
}

interface GoogleGlobal {
  accounts?: GoogleAccountsNamespace;
}

interface GoogleUserInfoResponse {
  sub: string;
  email: string;
  name: string;
  picture?: string;
}

interface StoredGoogleAuth {
  user: GoogleUser;
}

export interface GoogleUser {
  sub: string;
  email: string;
  name: string;
  picture?: string;
}

declare global {
  interface Window {
    google?: GoogleGlobal;
  }
}

function loadGoogleIdentityScript(): Promise<void> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("Google sign-in is only available in the browser."));
  }

  if (window.google?.accounts?.oauth2) {
    return Promise.resolve();
  }

  if (googleScriptLoadPromise) {
    return googleScriptLoadPromise;
  }

  googleScriptLoadPromise = new Promise((resolve, reject) => {
    const existingScript = document.getElementById(GOOGLE_SCRIPT_ID) as HTMLScriptElement | null;
    if (existingScript) {
      if (window.google?.accounts?.oauth2) {
        resolve();
        return;
      }
      existingScript.addEventListener("load", () => resolve(), { once: true });
      existingScript.addEventListener("error", () => reject(new Error("Failed to load Google Sign-In script.")), { once: true });
      return;
    }

    const script = document.createElement("script");
    script.id = GOOGLE_SCRIPT_ID;
    script.src = GOOGLE_SCRIPT_SRC;
    script.async = true;
    script.defer = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load Google Sign-In script."));
    document.head.appendChild(script);
  });

  return googleScriptLoadPromise;
}

function getStoredGoogleUser(): GoogleUser | null {
  if (typeof window === "undefined") return null;

  try {
    const raw = window.localStorage.getItem(GOOGLE_AUTH_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredGoogleAuth>;
    if (!parsed.user?.sub || !parsed.user.email || !parsed.user.name) {
      return null;
    }
    return parsed.user;
  } catch {
    return null;
  }
}

function persistGoogleUser(user: GoogleUser | null): void {
  if (typeof window === "undefined") return;

  if (!user) {
    window.localStorage.removeItem(GOOGLE_AUTH_STORAGE_KEY);
    return;
  }

  const payload: StoredGoogleAuth = { user };
  window.localStorage.setItem(GOOGLE_AUTH_STORAGE_KEY, JSON.stringify(payload));
}

async function fetchGoogleUserInfo(accessToken: string): Promise<GoogleUser> {
  const response = await fetch("https://www.googleapis.com/oauth2/v3/userinfo", {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });

  if (!response.ok) {
    throw new Error("Failed to fetch Google user profile.");
  }

  const payload = (await response.json()) as GoogleUserInfoResponse;
  if (!payload.sub || !payload.email || !payload.name) {
    throw new Error("Incomplete Google user profile response.");
  }

  return {
    sub: payload.sub,
    email: payload.email,
    name: payload.name,
    picture: payload.picture,
  };
}

export function useGoogleAuth() {
  const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ?? "";

  const [user, setUser] = useState<GoogleUser | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const storedUser = getStoredGoogleUser();
    if (storedUser) {
      setUser(storedUser);
    }
  }, []);

  useEffect(() => {
    if (!clientId) {
      setIsReady(false);
      return;
    }

    let active = true;
    loadGoogleIdentityScript()
      .then(() => {
        if (active) {
          setIsReady(true);
        }
      })
      .catch((scriptError) => {
        if (!active) return;
        setIsReady(false);
        setError(scriptError instanceof Error ? scriptError.message : "Failed to initialize Google Sign-In.");
      });

    return () => {
      active = false;
    };
  }, [clientId]);

  const login = useCallback(async () => {
    if (!clientId) {
      const message = "Google login is not configured. Set NEXT_PUBLIC_GOOGLE_CLIENT_ID first.";
      setError(message);
      return false;
    }

    setIsLoading(true);
    setError(null);

    try {
      await loadGoogleIdentityScript();
      const oauth2 = window.google?.accounts?.oauth2;

      if (!oauth2?.initTokenClient) {
        throw new Error("Google Sign-In SDK is unavailable.");
      }

      const tokenResponse = await new Promise<GoogleTokenResponse>((resolve, reject) => {
        const tokenClient = oauth2.initTokenClient({
          client_id: clientId,
          scope: "openid email profile",
          callback: (response: GoogleTokenResponse) => {
            if (response.error) {
              reject(new Error(response.error_description || response.error));
              return;
            }
            resolve(response);
          },
          error_callback: (sdkError: GoogleSdkError) => {
            reject(new Error(sdkError?.type || "Google popup was closed or blocked."));
          },
        });

        tokenClient.requestAccessToken({ prompt: "consent" });
      });

      if (!tokenResponse.access_token) {
        throw new Error("Google did not return an access token.");
      }

      const profile = await fetchGoogleUserInfo(tokenResponse.access_token);
      setUser(profile);
      setAccessToken(tokenResponse.access_token);
      persistGoogleUser(profile);

      return true;
    } catch (loginError) {
      const message = loginError instanceof Error ? loginError.message : "Google login failed.";
      setError(message);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [clientId]);

  const logout = useCallback(() => {
    if (accessToken && window.google?.accounts?.oauth2?.revoke) {
      window.google.accounts.oauth2.revoke(accessToken, () => undefined);
    }

    setUser(null);
    setAccessToken(null);
    setError(null);
    persistGoogleUser(null);
  }, [accessToken]);

  const isLoggedIn = useMemo(() => Boolean(user), [user]);

  return {
    user,
    isLoggedIn,
    isLoading,
    isReady,
    error,
    isConfigured: Boolean(clientId),
    login,
    logout,
  };
}
