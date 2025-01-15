#include <iostream>
using namespace std;
int main() {
  string a;
  int i;
  cin >> a;
  a[0] = toupper(a[0]);
  for (i = 0; i < a.size(); i++) {
    cout << a[i];
  }
}
