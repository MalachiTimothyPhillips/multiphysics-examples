// multiphys_mms_pnp_linear.hpp
#pragma once

#include <cmath>
#include "multiphys_mesh.hpp"

namespace tt::mms_pnp_linear {

inline SC pi() { return SC(3.141592653589793238462643383279502884); }

// Exact potential
inline SC phi_ex(SC x, SC y) { return std::sin(pi()*x)*std::sin(pi()*y); }

// Exact concentration (positive)
inline SC c_ex(SC x, SC y) { return SC(1.0) + SC(0.1)*std::sin(2*pi()*x)*std::sin(pi()*y); }

// Prescribed velocity field u(x,y) (choose constant here; you can change to variable)
inline SC ux(SC x, SC y) { (void)x; (void)y; return SC(1.0); }
inline SC uy(SC x, SC y) { (void)x; (void)y; return SC(0.5); }

// Derivatives for phi
inline SC dphi_dx(SC x, SC y) { return pi()*std::cos(pi()*x)*std::sin(pi()*y); }
inline SC dphi_dy(SC x, SC y) { return pi()*std::sin(pi()*x)*std::cos(pi()*y); }
inline SC lap_phi(SC x, SC y) { return -SC(2)*pi()*pi()*std::sin(pi()*x)*std::sin(pi()*y); }

// Derivatives for c
inline SC dc_dx(SC x, SC y) { return SC(0.1)*SC(2)*pi()*std::cos(2*pi()*x)*std::sin(pi()*y); }
inline SC dc_dy(SC x, SC y) { return SC(0.1)*pi()*std::sin(2*pi()*x)*std::cos(pi()*y); }

inline SC d2c_dx2(SC x, SC y) { return -SC(0.1)*SC(4)*pi()*pi()*std::sin(2*pi()*x)*std::sin(pi()*y); }
inline SC d2c_dy2(SC x, SC y) { return -SC(0.1)*pi()*pi()*std::sin(2*pi()*x)*std::sin(pi()*y); }
inline SC lap_c(SC x, SC y) { return d2c_dx2(x,y) + d2c_dy2(x,y); }

// MMS Poisson source: -div(eps grad phi) = rho_f + zF c
inline SC rho_f(SC x, SC y, SC eps, SC zF)
{
  return -eps*lap_phi(x,y) - zF*c_ex(x,y);
}

// MMS NP source for linear drift NP:
// div(-D grad c - alpha grad phi + u c) = s
// = -D Δc - alpha Δphi + div(u c)
// For constant u: div(u c)=u·∇c
inline SC np_source(SC x, SC y, SC D, SC alpha)
{
  const SC div_uc = ux(x,y)*dc_dx(x,y) + uy(x,y)*dc_dy(x,y);
  return -D*lap_c(x,y) - alpha*lap_phi(x,y) + div_uc;
}

} // namespace tt::mms_pnp_linear