/*=====TENSHI29=====*/
#include <iostream>
//#include<algorithm>
#include <string>
using namespace std;
int main() {
  int n, t;
  // char q[n];
  string q;
  cin >> n >> t;
  // for(int i=0;i<n;i++)
  cin >> q;
  while (t--) {
    for (int j = 0; j < n; j++) {
      if (q[j] == 'B' && q[j + 1] == 'G') {
        // swap(q[j],q[j+1]);
        q[j + 1] = 'B';
        q[j] = 'G';
        j++;
      }
    }
  }
  for (int i = 0; i < n; i++)
    cout << q[i];
  return 0;
}
