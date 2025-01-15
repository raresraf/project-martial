#include <iostream>
#include <string>

using namespace std;
void capitalize(string &);

int main() {
  string input;
  cin >> input;

  capitalize(input);
  cout << input << endl;
}

void capitalize(string &str) {
  if (islower(str.at(0)))
    str.at(0) = toupper(str.at(0));
}