#include <iostream>
using namespace std;
main() {
  long long int n;
  cin >> n;
  int count = 0;
  while (n > 0) {
    int x = n % 10;
    if (x == 4 || x == 7)
      count += 1;
    n = n / 10;
  }
  // cout<<"count "<<count<<endl;
  if (count == 4 || count == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
