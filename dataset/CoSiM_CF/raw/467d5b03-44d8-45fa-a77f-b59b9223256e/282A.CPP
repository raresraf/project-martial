#include <iostream>
#include <string>
using namespace std;

int main() {

  int n, x = 0;
  string operation;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> operation;
    if (operation == "++X" || operation == "X++")
      x++;
    else
      x--;
  }
  cout << x;

  system("pause");
  return 0;
}
