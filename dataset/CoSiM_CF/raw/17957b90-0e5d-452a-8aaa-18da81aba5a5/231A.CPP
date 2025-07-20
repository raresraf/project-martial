#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n; // Number of contests

  int flag;
  flag = 0; // Number of contests they are going to attend

  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c;
    if (a + b + c >= 2) {
      flag++;
    }
  }
  cout << flag;
  return 0;
}
