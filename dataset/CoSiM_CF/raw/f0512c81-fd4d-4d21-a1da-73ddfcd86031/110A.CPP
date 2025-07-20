#include <bits/stdc++.h>

using namespace std;

int main(void) {
  long long n;
  cin >> n;

  int c = 0;
  // bool a = true;
  for (; n != 0; n /= 10) {

    if (n % 10 == 4 || n % 10 == 7) {
      c++;
    }
  }

  if (c == 4 || c == 7)
    printf("YES\n");
  else
    printf("NO\n");

  return 0;
}
