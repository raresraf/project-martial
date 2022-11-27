#include <iostream>

using namespace std;

int main() {
  int melons;

  cin >> melons;
  if (melons <= 2 || melons % 2)
    cout << "NO"
         << "\n";
  else
    cout << "YES"
         << "\n";

  return 0;
}
