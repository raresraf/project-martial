#include <iostream>
#include <stdlib.h>

using namespace std;

int main() {
  string a, b;
  cin >> a >> b;
  for (int i = 0; i < a.length(); i++) {
    a[i] = (int(a[i] > 90) ? a[i] : char(int(a[i]) + 32));
    b[i] = (int(b[i] > 90) ? b[i] : char(int(b[i]) + 32));
    if (a[i] != b[i]) {
      cout << ((a[i] < b[i]) ? -1 : 1);
      exit(0);
    }
  }
  cout << 0;

  /*string a,b;
  cin>>a>>b;
  for(int i=0; i<a.length();i++)
  {

  }*/
}
