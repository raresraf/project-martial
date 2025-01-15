#include <iostream>
using namespace std;
int main() {

  int n, m, db = 0;
  cin >> n >> m;

  n % 2 == 0 ? db = (n / 2) *m : db = (n / 2) * m + (m / 2);
  cout << db;

  return 0;
}
