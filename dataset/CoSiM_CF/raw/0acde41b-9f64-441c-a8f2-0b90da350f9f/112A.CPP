#include <iostream>
#include <string>

using namespace std;

int main() {
  string inp_1, inp_2;
  cin >> inp_1 >> inp_2;

  for (int i = 0; i < inp_1.length(); i++) {
    inp_1[i] = tolower(inp_1[i]);
  }

  for (int i = 0; i < inp_1.length(); i++) {
    inp_2[i] = tolower(inp_2[i]);
  }

  cout << inp_1.compare(inp_2);

  return 0;
}