#include <bits/stdc++.h>
using namespace std;

#define fast                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);

int main() {
  fast

      int n;
  cin >> n;

  if (n == 2 || n % 2 != 0)
    cout << "NO";
  else
    cout << "YES";

  return 0;
}
