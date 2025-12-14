#include <cctype>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string in;
  cin >> in;
  for (int i = 0; i < in.size(); i++) {
    char c = tolower(in[i]);

    if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c != 'y')

    {
      cout << "." << c;
    }
    // cout<<in<<endl;
  }
  cout << endl;
  return 0;
}