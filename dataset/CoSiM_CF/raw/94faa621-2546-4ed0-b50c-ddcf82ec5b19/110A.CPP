#include <bits/stdc++.h>
int main() {
  long long int n, c = 0, g;
  std::cin >> n;
  while (n > 0) {
    g = n % 10;
    n /= 10;
    if (g == 7 || g == 4)
      c++;
  }
  std::cout << ((c == 7 || c == 4) ? "YES" : "NO");
}
