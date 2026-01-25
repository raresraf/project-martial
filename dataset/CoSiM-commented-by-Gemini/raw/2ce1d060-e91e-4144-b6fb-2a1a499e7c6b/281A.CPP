#include <iostream>
#include <string>
using namespace std;

int main() {
  string kata;
  cin >> kata;

  if ((kata[0] >= 'a') && (kata[0] <= 'z')) {
    kata[0] = kata[0] - 'a' + 'A';
  }
  cout << kata;

  return 0;
}
