#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, x = 0;
  cin >> n;

  string temp[n];
  for (int i = 0; i < n; i++)
    cin >> temp[i];

  for (int i = 0; i < n; i++) {
    if (temp[i].at(0) == '+' || temp[i].at(2) == '+')
      x = x + 1;
    if (temp[i].at(0) == '-' || temp[i].at(2) == '-')
      x = x - 1;
  }

  cout << x << endl;
  return 0;
}
