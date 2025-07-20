#include <bits/extc++.h>
#include <bits/stdc++.h>

#define for0(i, n) for (auto i = decltype(n)(0); i < n; ++i)
#define for1(i, n) for (auto i = decltype(n)(1); i <= n; ++i)
#define for0r(i, n) for (auto i = n - 1; i >= 0; --i)
#define for1r(i, n) for (auto i = n; i >= 1; --i)

using ll = long long;
using Vi = std::vector<int>;
using Vll = std::vector<ll>;
using Pii = std::pair<int, int>;
using Pll = std::pair<ll, ll>;

// -------------------------------------------------
int c, m, x;

bool Check(int a, int b) {
    if (a == b) return true;
    if (a == b - 1) return true;
    if (a == b + 1) return true;

    return false;
}

void Solve() {
    scanf("%d %d %d", &c, &m, &x);
    int d = std::min(c, m);
    c -= d;
    m -= d;
    x += c + m;
    if (d <= x) {
        printf("%d\n", d);
        return;
    }

    // d > x
    // if d - x == 1, then can't move any from d to x
    // if d - x == 2, then can move 1
    // if d - x == 3, then can move 1
    // if d - x == 1 + 3n, then can move n
    // if d - x == 2 + 3n, then can move n + 1
    int n2 = (d - x + 1) / 3;  // 1   2      +1 /3
    int ans1 = d - n2;
    int ans2 = x + 2 * n2;
    int ans = std::min(ans1, ans2);
    printf("%d\n", ans);
}
// -------------------------------------------------

int main() {
#ifndef ONLINE_JUDGE
    ::freopen("../input.txt", "r", stdin);
#endif

#if 1
    int t;
    scanf("%d", &t);
    for0(i, t) Solve();
#else
    Solve();
#endif

    return 0;
}
