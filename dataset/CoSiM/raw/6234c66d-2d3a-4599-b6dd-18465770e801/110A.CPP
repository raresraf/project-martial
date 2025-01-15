
#include <bits/stdc++.h>

using namespace std;

int n, cnt;

int main() {
  scanf("%d", &n);
  while (n > 0)
    cnt += ((n % 10) == 4 || (n % 10) == 7), n /= 10;
  cout << cnt;
  if (cnt == 4 || cnt == 7)
    puts("YES");
  else
    puts("NO");
}
