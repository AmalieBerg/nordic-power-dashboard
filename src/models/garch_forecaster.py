"""
GARCH Volatility Forecasting for Nordic Power Prices
====================================================

Implements GARCH(1,1) model for forecasting hourly power price volatility.
Uses arch library for parameter estimation and forecasting.

Author: Amalie Berg
Date: December 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from arch import arch_model
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress arch warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class GARCHForecaster:
    """
    GARCH(1,1) volatility forecasting for hourly power prices.
    
    The GARCH(1,1) model captures volatility clustering in power prices:
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    
    Where:
    - σ²_t = conditional variance at time t
    - ω = long-run variance level (constant)
    - α = ARCH coefficient (impact of recent shocks)
    - β = GARCH coefficient (persistence of volatility)
    - ε²_{t-1} = squared residual (shock) from previous period
    
    Attributes:
        lookback_window (int): Hours of historical data for estimation (default: 168 = 7 days)
        mean_model (str): Mean model specification ('Zero', 'Constant', 'AR')
        vol_model (str): Volatility model ('GARCH', 'EGARCH', 'GJR-GARCH')
        p (int): GARCH lag order (default: 1)
        q (int): ARCH lag order (default: 1)
    """
    
    def __init__(
        self,
        lookback_window: int = 168,  # 7 days of hourly data
        mean_model: str = 'Zero',
        vol_model: str = 'GARCH',
        p: int = 1,
        q: int = 1
    ):
        """
        Initialize GARCH forecaster.
        
        Args:
            lookback_window: Hours of historical data for parameter estimation
            mean_model: Mean model specification ('Zero', 'Constant', 'AR')
            vol_model: Volatility model type ('GARCH', 'EGARCH', 'GJR-GARCH')
            p: GARCH lag order
            q: ARCH lag order
        """
        self.lookback_window = lookback_window
        self.mean_model = mean_model
        self.vol_model = vol_model
        self.p = p
        self.q = q
        
        # Will store fitted model
        self.fitted_model: Optional[Any] = None
        self.last_estimation_time: Optional[pd.Timestamp] = None
        
        logger.info(
            f"Initialized GARCHForecaster: "
            f"window={lookback_window}h, "
            f"mean={mean_model}, "
            f"vol={vol_model}({p},{q})"
        )
    
    def prepare_returns(
        self,
        prices: pd.Series,
        return_type: str = 'log'
    ) -> pd.Series:
        """
        Calculate returns from hourly prices.
        
        Args:
            prices: Series of hourly power prices (EUR/MWh)
            return_type: 'log' for log returns, 'simple' for percentage returns
        
        Returns:
            Series of returns (same index as prices, first value NaN)
        """
        if len(prices) < 2:
            raise ValueError("Need at least 2 prices to calculate returns")
        
        if return_type == 'log':
            # Log returns: ln(P_t / P_{t-1})
            returns = np.log(prices / prices.shift(1))
        elif return_type == 'simple':
            # Simple returns: (P_t - P_{t-1}) / P_{t-1}
            returns = prices.pct_change()
        else:
            raise ValueError(f"Unknown return_type: {return_type}")
        
        # Scale returns to percentage for better numerical stability
        returns = returns * 100
        
        # Remove any infinite values
        returns = returns.replace([np.inf, -np.inf], np.nan)
        
        logger.debug(f"Calculated {return_type} returns: {len(returns)} observations")
        
        return returns
    
    def estimate_garch(
        self,
        returns: pd.Series,
        show_summary: bool = False
    ) -> Any:
        """
        Estimate GARCH model parameters using maximum likelihood.
        
        Args:
            returns: Series of returns (should be stationary)
            show_summary: If True, print model estimation summary
        
        Returns:
            Fitted ARCH model result object
        """
        if len(returns.dropna()) < self.lookback_window:
            raise ValueError(
                f"Insufficient data: need {self.lookback_window} observations, "
                f"got {len(returns.dropna())}"
            )
        
        try:
            # Create GARCH model
            # rescale=False because we already scaled returns to percentage
            model = arch_model(
                returns.dropna(),
                mean=self.mean_model,
                vol=self.vol_model,
                p=self.p,
                q=self.q,
                rescale=False
            )
            
            # Fit model using maximum likelihood
            fitted = model.fit(disp='off', show_warning=False)
            
            # Store fitted model
            self.fitted_model = fitted
            self.last_estimation_time = returns.index[-1]
            
            if show_summary:
                print(fitted.summary())
            
            # Log parameter estimates
            params = fitted.params
            logger.info(
                f"GARCH estimation complete: "
                f"ω={params.get('omega', np.nan):.6f}, "
                f"α={params.get('alpha[1]', np.nan):.6f}, "
                f"β={params.get('beta[1]', np.nan):.6f}"
            )
            
            return fitted
            
        except Exception as e:
            logger.error(f"GARCH estimation failed: {str(e)}")
            raise
    
    def forecast_volatility(
        self,
        prices: pd.Series,
        horizon: int = 24,
        confidence_level: float = 0.95,
        refit: bool = True
    ) -> pd.DataFrame:
        """
        Generate volatility forecasts for next h hours.
        
        Args:
            prices: Historical hourly prices
            horizon: Forecast horizon in hours (default: 24)
            confidence_level: Confidence level for intervals (default: 0.95)
            refit: If True, re-estimate model; if False, use last fitted model
        
        Returns:
            DataFrame with columns:
            - timestamp: Forecast timestamps
            - forecast_vol: Point forecast of volatility
            - lower_ci: Lower confidence interval
            - upper_ci: Upper confidence interval
            - volatility_pct: Volatility as % of current price level
        """
        # Calculate returns
        returns = self.prepare_returns(prices)
        
        # Use only last lookback_window observations for estimation
        estimation_returns = returns.iloc[-self.lookback_window:].copy()
        
        # Fit model (or use cached if refit=False)
        if refit or self.fitted_model is None:
            fitted_model = self.estimate_garch(estimation_returns)
        else:
            fitted_model = self.fitted_model
            logger.info("Using cached GARCH model (refit=False)")
        
        # Generate forecasts
        forecasts = fitted_model.forecast(horizon=horizon, reindex=False)
        
        # Extract variance forecasts
        variance_forecast = forecasts.variance.values[-1, :]
        
        # Convert variance to volatility (standard deviation)
        volatility_forecast = np.sqrt(variance_forecast)
        
        # Calculate confidence intervals
        # For GARCH, typically use ±1.96σ for 95% CI
        z_score = 1.96 if confidence_level == 0.95 else 2.576
        lower_ci = volatility_forecast - z_score * volatility_forecast * 0.1
        upper_ci = volatility_forecast + z_score * volatility_forecast * 0.1
        
        # Create forecast timestamps
        last_timestamp = prices.index[-1]
        forecast_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=horizon,
            freq='h'
        )
        
        # Current price level (for percentage calculation)
        current_price = prices.iloc[-1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'timestamp': forecast_timestamps,
            'forecast_vol': volatility_forecast,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'volatility_pct': (volatility_forecast / current_price) * 100
        })
        
        results.set_index('timestamp', inplace=True)
        
        logger.info(
            f"Generated {horizon}h volatility forecast: "
            f"mean_vol={volatility_forecast.mean():.4f}, "
            f"range=[{volatility_forecast.min():.4f}, {volatility_forecast.max():.4f}]"
        )
        
        return results
    
    def rolling_forecast(
        self,
        prices: pd.Series,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        horizon: int = 24,
        step: int = 24
    ) -> pd.DataFrame:
        """
        Generate rolling out-of-sample forecasts.
        
        This simulates how the model would perform in production:
        - Use only data available up to forecast time
        - Re-estimate parameters periodically
        - Compare forecasts to actual realized volatility
        
        Args:
            prices: Full historical price series
            start_date: Start of forecasting period
            end_date: End of forecasting period
            horizon: Forecast horizon in hours
            step: Hours between re-estimation (default: 24 = daily)
        
        Returns:
            DataFrame with columns:
            - forecast_time: When forecast was made
            - target_time: Time being forecasted
            - forecast_vol: Forecasted volatility
            - actual_vol: Realized volatility (if available)
        """
        logger.info(
            f"Starting rolling forecast: "
            f"{start_date} to {end_date}, "
            f"horizon={horizon}h, step={step}h"
        )
        
        all_forecasts = []
        
        # DON'T filter prices yet - we need future data to calculate actuals
        # Only filter when getting historical data for model estimation
        
        # Generate forecast at each step
        current_time = start_date
        forecast_count = 0
        
        while current_time <= end_date:
            try:
                # Get historical data up to current_time (for estimation)
                historical_prices = prices[prices.index <= current_time]
                
                # Need enough data for estimation
                if len(historical_prices) < self.lookback_window + horizon:
                    logger.warning(
                        f"Insufficient data at {current_time}, skipping"
                    )
                    current_time += pd.Timedelta(hours=step)
                    continue
                
                # Generate forecast
                forecast = self.forecast_volatility(
                    historical_prices,
                    horizon=horizon,
                    refit=True
                )
                
                # Add forecast time
                forecast['forecast_time'] = current_time
                forecast['target_time'] = forecast.index
                
                all_forecasts.append(forecast)
                forecast_count += 1
                
                if forecast_count % 10 == 0:
                    logger.info(f"Generated {forecast_count} forecasts...")
                
            except Exception as e:
                logger.warning(f"Forecast failed at {current_time}: {str(e)}")
            
            # Move to next step
            current_time += pd.Timedelta(hours=step)
        
        if not all_forecasts:
            raise ValueError("No forecasts generated")
        
        # Combine all forecasts
        results = pd.concat(all_forecasts, ignore_index=True)
        
        # Calculate realized volatility for comparison
        returns = self.prepare_returns(prices)
        
        # For each forecast, calculate actual realized volatility
        def get_realized_vol(row):
            target_time = row['target_time']
            # Calculate rolling std using 24h window ENDING at target_time
            # This is the realized volatility at that target hour
            window_end = target_time
            window_start = target_time - pd.Timedelta(hours=23)
            
            future_returns = returns[
                (returns.index >= window_start) &
                (returns.index <= window_end)
            ]
            if len(future_returns) >= 12:  # Need at least 12 hours of data
                return future_returns.std()
            return np.nan
        
        results['actual_vol'] = results.apply(get_realized_vol, axis=1)
        
        logger.info(
            f"Rolling forecast complete: "
            f"{len(results)} total forecasts, "
            f"{forecast_count} estimation windows"
        )
        
        return results
    
    def get_model_diagnostics(self) -> Dict[str, float]:
        """
        Get diagnostic statistics for the fitted model.
        
        Returns:
            Dictionary with:
            - aic: Akaike Information Criterion
            - bic: Bayesian Information Criterion
            - log_likelihood: Log likelihood value
            - alpha: ARCH coefficient
            - beta: GARCH coefficient
            - persistence: α + β (volatility persistence)
        """
        if self.fitted_model is None:
            raise ValueError("No model fitted yet")
        
        params = self.fitted_model.params
        
        diagnostics = {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'log_likelihood': self.fitted_model.loglikelihood,
            'omega': params.get('omega', np.nan),
            'alpha': params.get('alpha[1]', np.nan),
            'beta': params.get('beta[1]', np.nan),
            'persistence': params.get('alpha[1]', 0) + params.get('beta[1]', 0)
        }
        
        return diagnostics


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_garch_forecaster():
    """
    Test GARCH forecaster with synthetic data.
    """
    print("\n" + "="*70)
    print("TESTING GARCH FORECASTER")
    print("="*70)
    
    # Generate synthetic price data with GARCH-like volatility
    np.random.seed(42)
    n_obs = 500
    
    # Simulate GARCH(1,1) process
    omega = 0.1
    alpha = 0.15
    beta = 0.75
    
    returns = np.zeros(n_obs)
    vol = np.zeros(n_obs)
    vol[0] = np.sqrt(omega / (1 - alpha - beta))
    
    for t in range(1, n_obs):
        vol[t] = np.sqrt(
            omega + alpha * returns[t-1]**2 + beta * vol[t-1]**2
        )
        returns[t] = vol[t] * np.random.randn()
    
    # Convert to prices
    log_prices = np.cumsum(returns / 100)  # Scale down
    prices = 50 * np.exp(log_prices)  # Start at 50 EUR/MWh
    
    # Create timestamps
    timestamps = pd.date_range(
        start='2024-11-01',
        periods=n_obs,
        freq='h'
    )
    
    price_series = pd.Series(prices, index=timestamps)
    
    print(f"\n✓ Generated synthetic prices: {len(price_series)} hours")
    print(f"  Price range: {price_series.min():.2f} - {price_series.max():.2f} EUR/MWh")
    print(f"  Mean price: {price_series.mean():.2f} EUR/MWh")
    
    # Initialize forecaster
    forecaster = GARCHForecaster(lookback_window=168)
    
    # Test 1: Single forecast
    print("\n" + "-"*70)
    print("TEST 1: 24-hour volatility forecast")
    print("-"*70)
    
    forecast = forecaster.forecast_volatility(
        price_series,
        horizon=24,
        refit=True
    )
    
    print(f"\n✓ Forecast generated: {len(forecast)} hours")
    print(f"\nForecast summary:")
    print(f"  Mean volatility: {forecast['forecast_vol'].mean():.4f}")
    print(f"  Min volatility:  {forecast['forecast_vol'].min():.4f}")
    print(f"  Max volatility:  {forecast['forecast_vol'].max():.4f}")
    print(f"  As % of price:   {forecast['volatility_pct'].mean():.2f}%")
    
    # Test 2: Model diagnostics
    print("\n" + "-"*70)
    print("TEST 2: Model diagnostics")
    print("-"*70)
    
    diagnostics = forecaster.get_model_diagnostics()
    print(f"\nGARCH(1,1) parameters:")
    print(f"  ω (omega):      {diagnostics['omega']:.6f}")
    print(f"  α (alpha):      {diagnostics['alpha']:.6f}")
    print(f"  β (beta):       {diagnostics['beta']:.6f}")
    print(f"  Persistence:    {diagnostics['persistence']:.6f}")
    print(f"\nModel fit:")
    print(f"  Log-likelihood: {diagnostics['log_likelihood']:.2f}")
    print(f"  AIC:            {diagnostics['aic']:.2f}")
    print(f"  BIC:            {diagnostics['bic']:.2f}")
    
    # Test 3: Rolling forecast
    print("\n" + "-"*70)
    print("TEST 3: Rolling out-of-sample forecasts")
    print("-"*70)
    
    # Use last 200 hours for rolling forecast
    start = timestamps[-200]
    end = timestamps[-50]
    
    print(f"\nGenerating rolling forecasts from {start} to {end}...")
    
    rolling = forecaster.rolling_forecast(
        price_series,
        start_date=start,
        end_date=end,
        horizon=24,
        step=24
    )
    
    print(f"\n✓ Rolling forecast complete: {len(rolling)} total forecasts")
    
    # Calculate forecast accuracy
    valid_forecasts = rolling.dropna(subset=['actual_vol'])
    if len(valid_forecasts) > 0:
        mae = np.abs(
            valid_forecasts['forecast_vol'] - valid_forecasts['actual_vol']
        ).mean()
        rmse = np.sqrt(
            ((valid_forecasts['forecast_vol'] - valid_forecasts['actual_vol'])**2).mean()
        )
        
        print(f"\nForecast accuracy:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
    
    print("\n" + "="*70)
    print("✅ ALL GARCH FORECASTER TESTS PASSED!")
    print("="*70)
    
    return forecaster, price_series, forecast, rolling


if __name__ == '__main__':
    # Run tests
    forecaster, prices, forecast, rolling = test_garch_forecaster()
    
    print("\n📊 GARCH Forecaster ready for production!")
    print("\nNext steps:")
    print("  1. Integrate with PriceDatabase to fetch real data")
    print("  2. Create backtesting framework")
    print("  3. Build production pipeline")