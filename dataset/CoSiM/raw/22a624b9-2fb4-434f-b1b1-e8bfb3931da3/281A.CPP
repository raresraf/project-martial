#include <iostream>
using namespace std;
int main() {

  string txt;
  cin >> txt;
  txt[0] = toupper(txt[0]);
  cout << txt;

  return 0;
}
