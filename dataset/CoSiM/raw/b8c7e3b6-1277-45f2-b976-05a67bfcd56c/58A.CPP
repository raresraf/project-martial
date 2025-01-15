#include <iostream>
#include <string>
using namespace std;
int main() {
  string in, comp = "hello";
  cin >> in;
  int a = 0;
  for (int i = 0; i < in.length(); i++) {
    if (in[i] == comp[a]) {
      a++;
    }
  }
  if (a == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
