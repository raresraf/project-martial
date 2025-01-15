#include <bits/stdc++.h>

using namespace std;

int main() {
  string str;
  int value = 0, n, i;
  scanf("%d\n", &n);
  for (i = 1; i <= n; i++) {
    cin >> str;
    if (str == "X++" || str == "++X") {
      value++;
    } else {
      value--;
    }
  }
  cout << value << endl;
}
