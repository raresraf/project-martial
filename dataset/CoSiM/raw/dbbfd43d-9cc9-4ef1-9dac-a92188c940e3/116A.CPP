#include <bits/stdc++.h>
using namespace std;

int main() {
  int in, out, n, rem = 0, max = 0;

  cin >> n;

  while (n--) {
    scanf("%d %d", &out, &in);
    rem = rem - out + in;
    if (rem > max)
      max = rem;
  }

  cout << max << endl;

  return 0;
}
