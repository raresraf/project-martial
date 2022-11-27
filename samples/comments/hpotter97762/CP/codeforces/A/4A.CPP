#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  if (n % 2 || n == 2)
    cout << "No\n";
  else
    cout << "Yes\n";
  return 0;
}