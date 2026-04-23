import globals from 'globals';
import pluginJs from '@eslint/js';
import tseslint from 'typescript-eslint';
import eslintPluginPrettierRecommended from 'eslint-plugin-prettier/recommended';

/** @type {import('eslint').Linter.Config[]} */
export default [
  {
    ignores: [
      '**/dist/**',
      '**/node_modules/**',
      '**/.worktrees/**',
      '**/coverage/**'
    ]
  },
  {
    files: ['**/*.{js,mjs,cjs,ts}'],
    languageOptions: { ecmaVersion: 2020, globals: globals.browser }
  },
  pluginJs.configs.recommended,
  ...tseslint.configs.recommended,
  eslintPluginPrettierRecommended,
  {
    name: 'Custom Rules',
    rules: {
      // Helps with cleaning debug statements by erroring on console.
      'no-console': 'error',
      // Disable unused vars, handles TS-specific cases (type params,
      // interfaces) better than base rule.
      // https://typescript-eslint.io/rules/no-unused-vars/#what-benefits-does-this-rule-have-over-typescript
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          args: 'all',
          argsIgnorePattern: '^_',
          caughtErrors: 'all',
          caughtErrorsIgnorePattern: '^_',
          destructuredArrayIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          ignoreRestSiblings: true
        }
      ],
      // Warn when `any` type is used. It's sometimes necessary, but should be
      // avoided when possible.
      '@typescript-eslint/no-explicit-any': 'warn'
    }
  }
];
