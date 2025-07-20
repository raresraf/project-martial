#include <iostream>
using namespace std;
int main() {

  int n;
  string txt;
  int db = 0;
  cin >> n >> txt;

  for (int i = 0; i < n; i++) {
    txt[i] == txt[i + 1] ? db++ : db;
  }

  cout << db;

  return 0;
}
