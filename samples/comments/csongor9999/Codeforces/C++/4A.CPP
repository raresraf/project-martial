#include <iostream>
using namespace std;
int main() {

  int n;
  cin >> n;

  n % 2 == 0 && n != 2 ? cout << "YES" : cout << "NO";

  return 0;
}
