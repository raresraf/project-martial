#include <bits/stdc++.h>

using namespace std;

int main() {
  long long int n, i;
  int count = 0;
  char str[100];

  cin >> n;
  sprintf(str, "%lld", n);

  for (i = 0; str[i] != '\0'; i++) {

    if (str[i] == '4' || str[i] == '7') {
      count++;
    }
  }

  if (count == 4 || count == 7) {
    cout << "YES\n";
  } else {
    cout << "NO\n";
  }
}
