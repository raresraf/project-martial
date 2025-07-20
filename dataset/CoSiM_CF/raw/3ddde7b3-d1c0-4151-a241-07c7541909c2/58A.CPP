#include <iostream>
using namespace std;
int main() {
  string a;
  string b = "hello";
  cin >> a;
  int i, j = 0, k = 0;
  for (i = 0; i < a.size(); i++) {
    if (a[i] == b[k]) {
      k++;
      j++;
    }
  }
  if (j == 5) {
    cout << "YES";
  } else
    cout << "NO";
}
