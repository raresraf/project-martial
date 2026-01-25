#include <algorithm>
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  char a[110];

  cin >> a;
  int n = strlen(a);
  sort(a, a + n);
  for (int i = 0; i < n; i++) {
    if (a[i] > 43) {
      cout << a[i];
      if (i + 1 < n)
        cout << "+";
    }
  }
}
