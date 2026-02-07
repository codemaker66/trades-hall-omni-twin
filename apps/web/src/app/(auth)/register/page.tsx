'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../hooks/useAuth'
import { Button } from '../../components/ui/Button'
import { Input } from '../../components/ui/Input'

export default function RegisterPage() {
  const { register } = useAuth()
  const router = useRouter()
  const [email, setEmail] = useState('')
  const [name, setName] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    const result = await register(email, name, password)
    setLoading(false)

    if (result.ok) {
      router.push('/')
    } else {
      setError(result.error ?? 'Registration failed.')
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-surface-0">
      <div className="w-full max-w-sm p-8 border border-surface-25 rounded-sm bg-surface-10">
        <h1 className="text-xl font-semibold text-surface-90 mb-6 text-center">
          Create your account
        </h1>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <Input
            label="Name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            autoComplete="name"
          />

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
            autoComplete="new-password"
            minLength={8}
          />

          {error && (
            <p className="text-danger-50 text-sm">{error}</p>
          )}

          <Button type="submit" disabled={loading}>
            {loading ? 'Creating account...' : 'Create account'}
          </Button>
        </form>

        <p className="text-surface-60 text-sm mt-4 text-center">
          Already have an account?{' '}
          <a href="/login" className="text-accent-60 hover:text-accent-70 underline">
            Sign in
          </a>
        </p>
      </div>
    </div>
  )
}
