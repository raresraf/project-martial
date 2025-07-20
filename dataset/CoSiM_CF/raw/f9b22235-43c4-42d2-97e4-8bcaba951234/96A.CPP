
#include <bits/stdc++.h>

using namespace std;

string s;
int now[2], mx[2], last = -1, a[123];

int main() {
  cin >> s;
  int n = (int)(s.length());
  for (int i = 0; i < n; i++)
    a[i] = (s[i] == '1');
  for (int i = 0; i < n; i++) {
    if (a[i])
      now[1]++;
    else
      mx[1] = max(mx[1], now[1]), now[1] = 0;
    if (!a[i])
      now[0]++;
    else
      mx[0] = max(mx[0], now[0]), now[0] = 0;
  }
  mx[0] = max(mx[0], now[0]);
  mx[1] = max(mx[1], now[1]);
  if (mx[0] > 6 || mx[1] > 6)
    puts("YES");
  else
    puts("NO");
}
