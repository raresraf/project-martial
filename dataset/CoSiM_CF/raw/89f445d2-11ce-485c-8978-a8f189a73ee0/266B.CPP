#include <iostream>
using namespace std;
main() {
  int n, t;
  cin >> n >> t;
  string str;
  cin >> str;
  while (t > 0) {
    for (int i = 0; i < n - 1; i++) {
      if (str[i] == 'B' && str[i + 1] == 'G') {
        swap(str[i], str[i + 1]);
        // cout<<"each "<<str<<endl;
        i += 1;
      }
    }
    t -= 1;
  }
  cout << str << endl;
}
