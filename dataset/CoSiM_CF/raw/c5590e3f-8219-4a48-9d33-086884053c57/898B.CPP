#include <bits/stdc++.h>

using namespace std;

// ax + by = c
pair<long long, long long> solve(int a, int b) {
  if (b == 0) {
    return {1, 0};
  }
  long long x, y;
  tie(x, y) = solve(b, a % b);
  long long q = a / b;
  // a = q * b + (a % b = r)
  // r = a - q * b
  // ...
  // eu sei que
  // g = b * x + r * y
  // ...
  // g = b * x + (a - q * b) * y
  // g = a * y + b * (x - q * y)
  long long nx = y;
  long long ny = x - q * y;
  return {nx, ny};
}

int main() {
  long long n, a, b;
  cin >> n >> a >> b;
  long long g = __gcd(a, b);
  if (n % g) {
    puts("NO");
    return 0;
  }
  a /= g, b /= g, n /= g;
  long long x, y;
  tie(x, y) = solve(a, b);
  x *= n;
  y *= n;
  // sx = x + t * b >= 0, sy = y - t * a >= 0
  // -x / b <= t , t <= y / a
  // -x / b <= y / a
  // ceil(-x / b) <= floor(y / a)
  long long lt = ceil(-x / double(b)), ht = floor(y / double(a));
  if (lt <= ht) {
    puts("YES");
    cout << (x + lt * b) << " " << (y - a * lt) << endl;
  } else {
    puts("NO");
  }
}
