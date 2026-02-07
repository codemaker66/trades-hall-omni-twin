'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../hooks/useAuth'
import { Button } from '../../components/ui/Button'
import { Input } from '../../components/ui/Input'

export default function LoginPage() {
  const { login } = useAuth()
  const router = useRouter()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    const result = await login(email, password)
    setLoading(false)

    if (result.ok) {
      router.push('/')
    } else {
      setError(result.error ?? 'Login failed.')
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-surface-0">
      <div className="w-full max-w-sm p-8 border border-surface-25 rounded-sm bg-surface-10">
        <h1 className="text-xl font-semibold text-surface-90 mb-6 text-center">
          Sign in to OmniTwin
        </h1>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <Input
            label="Email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            autoComplete="email"
          />

          <Input
            label="Password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            autoComplete="current-password"
          />

          {error && (
            <p className="text-danger-50 text-sm">{error}</p>
          )}

          <Button type="submit" disabled={loading}>
            {loading ? 'Signing in...' : 'Sign in'}
          </Button>
        </form>

        <p className="text-surface-60 text-sm mt-4 text-center">
          Don&apos;t have an account?{' '}
          <a href="/register" className="text-accent-60 hover:text-accent-70 underline">
            Register
          </a>
        </p>
      </div>
    </div>
  )
}
