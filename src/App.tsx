import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AppStateProvider } from './contexts/AppStateContext';
import { ConfigProvider } from './contexts/ConfigContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { ToastProvider } from './contexts/ToastContext';
import { ConfirmProvider } from './contexts/ConfirmContext';
import { AppShell } from './components/layout';
import { Welcome } from './pages/Welcome';
import { Settings } from './pages/Settings';
import { SystemInfo } from './pages/SystemInfo';
import { Models } from './pages/Models';
import { LoRA } from './pages/LoRA';
import { Launch } from './pages/Launch';
import { Browse } from './pages/Browse';
import { Storage } from './pages/Storage';

function App() {
  return (
    <AppStateProvider>
      <ConfigProvider>
        <ThemeProvider>
          <ToastProvider>
            {/* Inside ToastProvider so a confirmed action can raise a toast,
              * and outside the router so a question survives a route change. */}
            <ConfirmProvider>
            <BrowserRouter>
              <Routes>
                <Route path="/" element={<AppShell />}>
                  <Route index element={<Welcome />} />
                  <Route path="launch" element={<Launch />} />
                  <Route path="browse" element={<Browse />} />
                  <Route path="settings" element={<Settings />} />
                  <Route path="system" element={<SystemInfo />} />
                  {/* Storage manages what is on disk; Discover finds new models.
                    * The old combined page is kept at /models-legacy while its
                    * download flow is folded into Discover. */}
                  <Route path="models" element={<Storage />} />
                  <Route path="models-legacy" element={<Models />} />
                  <Route path="lora" element={<LoRA />} />
                </Route>
              </Routes>
            </BrowserRouter>
            </ConfirmProvider>
          </ToastProvider>
        </ThemeProvider>
      </ConfigProvider>
    </AppStateProvider>
  );
}

export default App;