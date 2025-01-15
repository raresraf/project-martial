#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;

  int counter = 0;
  while (n--) {
    int temp = 0;
    for (int i = 0; i < 3; i++) {

      int a;
      cin >> a;
      if (a == 1)
        temp++;
    }
    if (temp >= 2)
      counter++;
  }
  cout << counter << "\n";
  return 0;
}
